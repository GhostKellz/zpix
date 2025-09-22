const std = @import("std");
const zpix = @import("zpix");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Create a simple 2x2 RGB image
    var image = try zpix.Image.init(allocator, 2, 2, .rgb);
    defer image.deinit();

    // Fill with red pixels: R=255, G=0, B=0
    for (0..4) |i| {
        image.data[i * 3] = 255; // R
        image.data[i * 3 + 1] = 0; // G
        image.data[i * 3 + 2] = 0; // B
    }

    // Save as BMP
    try image.save("/tmp/test.bmp", .bmp);
    std.debug.print("Image saved to /tmp/test.bmp\n", .{});

    // Test JPEG loading (if file exists)
    const jpeg_path = "/data/projects/zpix/file_example_JPG_100kB.jpg";
    if (std.fs.openFileAbsolute(jpeg_path, .{})) |jf| {
        jf.close();
        var jpeg_image = try zpix.Image.load(allocator, jpeg_path);
        defer jpeg_image.deinit();
        try jpeg_image.save("/tmp/test_loaded.jpg.bmp", .bmp);
        std.debug.print("JPEG loaded and saved as BMP\n", .{});
    } else |_| {
        std.debug.print("No JPEG file found at {s}\n", .{jpeg_path});
    }

    // Test resize
    // try image.resize(4, 4);
    // try image.save("/tmp/test_resized.bmp", .bmp);
    // std.debug.print("Resized image saved to /tmp/test_resized.bmp\n", .{});

    // Test crop
    // try image.crop(1, 1, 2, 2);
    // try image.save("/tmp/test_cropped.bmp", .bmp);
    // std.debug.print("Cropped image saved to /tmp/test_cropped.bmp\n", .{});

    // Test rotate
    // try image.rotate(90);
    // try image.save("/tmp/test_rotated.bmp", .bmp);
    // std.debug.print("Rotated image saved to /tmp/test_rotated.bmp\n", .{});

    // Test convert to grayscale
    // try image.convert(.grayscale);
    // try image.save("/tmp/test_gray.bmp", .bmp);
    // std.debug.print("Grayscale image saved to /tmp/test_gray.bmp\n", .{});

    // Test brightness
    // try image.brightness(50);
    // try image.save("/tmp/test_bright.bmp", .bmp);
    // std.debug.print("Brightened image saved to /tmp/test_bright.bmp\n", .{});

    // Test contrast
    // try image.contrast(1.2);
    // try image.save("/tmp/test_contrast.bmp", .bmp);
    // std.debug.print("Contrasted image saved to /tmp/test_contrast.bmp\n", .{});

    // Test blur
    // try image.blur(1);
    // try image.save("/tmp/test_blur.bmp", .bmp);
    // std.debug.print("Blurred image saved to /tmp/test_blur.bmp\n", .{});
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa); // Try commenting this out and see if zig detects the memory leak!
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
